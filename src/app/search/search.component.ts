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
	searchText: string = "obamacare";
	tfidfResults$: Observable<any>;
	secondaryIndexResults$: Observable<any>;

	searchTypes: object[] = [
		{ 'label': 'Article Content', 'value': 'content' },
		{ 'label': 'Article Title', 'value': 'title' },
	];
	searchType: string;
	@Output("results") searchResultsChange = new EventEmitter<any>();
  getTopKResults() {
		let queryParams = [`q=${this.searchText}`];
		if (this.searchType) {
			queryParams.push(`type=${this.searchType}`);
		}
		let queryString = queryParams.join("&");
		this.getTopKTfidf(queryString);
		this.getTopKSecondaryIndex(queryString);
		let allSearchResults = {
			"tfidf": this.tfidfResults$,
			"secondary_index": this.secondaryIndexResults$
		}
		this.searchResultsChange.emit(allSearchResults)
		return
	}
	getTopKTfidf(queryString: string) {
		this.tfidfResults$ = this.http.get<any>(`http://localhost:4200/api/documents-tfidf?${queryString}`).pipe(
			shareReplay({refCount: true, bufferSize:1})
		);
		return;
	}
	getTopKSecondaryIndex(queryString: string) {
		this.secondaryIndexResults$ = this.http.get<any>(`http://localhost:4200/api/documents-si?${queryString}`).pipe(
			shareReplay({refCount: true, bufferSize:1})
		)
		return;
	}
	ngOnInit(): void {

	}

}
