import { Component, OnInit, Input, Output, EventEmitter, HostListener } from '@angular/core';
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
	ngOnInit(): void {

	}
	searchText: string = "";
	searchQuantity: number = 5;
	MIN_SEARCH_QUANTITY: number = 1;
	MAX_SEARCH_QUANTITY: number = 100;

	tfidfResults$: Observable<any>;
	secondaryIndexResults$: Observable<any>;

	searchTypes: object[] = [
		{ 'label': 'Article Content', 'value': 'content' },
		{ 'label': 'Article Title', 'value': 'title' },
	];
	searchType: string = this.searchTypes[0]["value"];
	@Output("results") searchResultsChange = new EventEmitter<any>();
  getTopKResults() {
		if (this.searchText != "" && this.searchText != undefined && this.searchText != null
		&& !this.boundError()) {
			let queryParams = [`q=${this.searchText}`, `t=${this.searchType}`, `n=${this.searchQuantity}`];
			let queryString = queryParams.join("&");
			this.getTopKTfidf(queryString);
			this.getTopKSecondaryIndex(queryString);
			// Emit observables and subscribe on the results component using async pipe
			let allSearchResults = {
				"tfidf": this.tfidfResults$,
				"secondary_index": this.secondaryIndexResults$
			}
			this.searchResultsChange.emit(allSearchResults)
		}
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
	boundError() {
		return this.searchQuantity == null || this.searchQuantity == undefined || this.searchQuantity < this.MIN_SEARCH_QUANTITY || this.searchQuantity > this.MAX_SEARCH_QUANTITY
	}
	@HostListener('document:keydown', ['$event'])
	handleEnter(event:KeyboardEvent) {
		if (event.key == "Enter") { this.getTopKResults() }
	}
}
