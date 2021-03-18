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
	@Output("results") searchResultsChange = new EventEmitter<any>();
  getTopKResults() {
		this.searchResults = []
		this.items$ = this.http.get<any>(`http://localhost:4200/api/document?q=${this.searchText}`).pipe(
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
