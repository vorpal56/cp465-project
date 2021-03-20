import { Component, OnInit, Input, Output, EventEmitter } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { shareReplay } from 'rxjs/operators';

@Component({
  selector: 'search-bar',
  templateUrl: './search.component.html',
  styleUrls: ['./search.component.scss']
})
export class SearchComponent implements OnInit {

  constructor(private http: HttpClient) { }
	searchText: string = "";
	searchResults: any[] = [];
	items$: Observable<any>;

	searchTypes: object[] = [
		// { 'label': 'Both', 'value': '' },
		{ 'label': 'Article Content', 'value': 'content' },
		{ 'label': 'Article Title', 'value': 'title' },
	]
	searchType: string;
	@Output("results") searchResultsChange = new EventEmitter<any>();
  getTopKResults() {
		this.searchResults = []
		let queryParams = [`q=${this.searchText}`]
		if (this.searchType) {
			queryParams.push(`type=${this.searchType}`)
		}
		let queryString = queryParams.join("&")
		this.items$ = this.http.get<any>(`http://localhost:4200/api/document?${queryString}`).pipe(
			shareReplay({refCount: true, bufferSize:1})
		)
		this.items$.subscribe((results: any[]) => {
			results.forEach((result: any) => {
				this.searchResults.push(result);
			})
		})
		this.searchResultsChange.emit(this.searchResults)
	}
	ngOnInit(): void {

	}

}
